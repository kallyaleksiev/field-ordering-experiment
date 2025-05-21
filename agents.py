import os

# import logfire
from pydantic import BaseModel
from pydantic_ai import Agent

from prep_task import AUXILIARY_LS, LABEL_NAMES

# logfire.configure()
# logfire.instrument_pydantic_ai()

MODEL_NAME = os.getenv("MODEL_NAME", "openai:gpt-4.1-mini")


class EasyTaskAgentOutputAnswerFirst(BaseModel):
    label: int
    reasoning: str


class EasyTaskAgentOutputAnswerSecond(BaseModel):
    reasoning: str
    label: int


class HardTaskAgentOutputAnswerFirst(BaseModel):
    answer: str
    reasoning: str


class HardTaskAgentOutputAnswerSecond(BaseModel):
    reasoning: str
    answer: str


et_af_agent = Agent(
    model=MODEL_NAME,
    output_type=EasyTaskAgentOutputAnswerFirst,
)
et_as_agent = Agent(
    model=MODEL_NAME,
    output_type=EasyTaskAgentOutputAnswerSecond,
)

ht_af_agent = Agent(
    model=MODEL_NAME,
    output_type=HardTaskAgentOutputAnswerFirst,
)
ht_as_agent = Agent(
    model=MODEL_NAME,
    output_type=HardTaskAgentOutputAnswerSecond,
)

_label_str = "\n".join([
    f"Index: {i}, Label String: {name.replace('_', ' ')}" for i, name in enumerate(LABEL_NAMES)
])


@ht_af_agent.system_prompt
def ht_af_system_prompt() -> str:
    return f"""
    You are a helpful assistant.

    You first assign each image the number of the label that best describes the style of the painting by choosing from the following list:

    {_label_str}

    Then you provide as an answer the string that has exactly i+2 letters where i is the index of the label you chose in the previous step.

    The answer should be selected from the following list:

    {AUXILIARY_LS}

    Provide your answer as a structured response in the following format:
    {HardTaskAgentOutputAnswerFirst.model_json_schema()}
    """


@ht_as_agent.system_prompt
def ht_as_system_prompt() -> str:
    return f"""
    You are a helpful assistant.

    You first assign each image the number of the label that best describes the style of the painting by choosing from the following list:

    {_label_str}

    Then you provide as an answer the string that has exactly i+2 letters where i is the index of the label you chose in the previous step.

    The answer should be selected from the following list:

    {AUXILIARY_LS}

    Provide your answer as a structured response in the following format:
    {HardTaskAgentOutputAnswerSecond.model_json_schema()}
    """


@et_as_agent.system_prompt
def et_as_system_prompt() -> str:
    return f"""
    You are a helpful assistant that classifies images. You assign each image the number of the label that best describes the style of the painting by choosing from the following list:

    {_label_str}

    Provide your answer as a structured response in the following format:
    {EasyTaskAgentOutputAnswerSecond.model_json_schema()}

    where the answer is an int.
    """


@et_af_agent.system_prompt
def et_af_system_prompt() -> str:
    return f"""
    You are a helpful assistant that classifies images. You assign each image the number of the label that best describes the style of the painting by choosing from the following list:

    {_label_str}

    Provide your answer as a structured response in the following format:
    {EasyTaskAgentOutputAnswerFirst.model_json_schema()}

    where the answer is an int.
    """
