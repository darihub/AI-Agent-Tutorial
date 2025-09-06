# Python
import os
from dotenv import load_dotenv
# Pydantic
from pydantic import BaseModel
# Langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import GPT4All
# GPT4All
from gpt4all import GPT4All as GPT4AllModel

# TODO: Change Agent's instructions to cover something else not related to the POS system.
# Make it connect to Wikpedia API
# Host GPT4All on a server and connect to it from here


load_dotenv() # Load environment variables from a .env file

# LLM Setup

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse) # parses raw LLM output into structured ResearchResponse

MODEL_PATH = os.getenv("MODEL_PATH")  # Update this path to your actual model file
llm = GPT4All(MODEL_PATH) # local model instance

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are a research assistant that answers questions based on Wikipedia.
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) # partially fill prompt with format instructions

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = []
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)
raw_response = agent_executor.invoke({"query": "Cómo añado un nuevo producto?"})
print(raw_response)

output = raw_response.get("output")
if output is not None:
    # Safe to subscript or process output
    first_item = output[0]  # Example
else:
    # Handle missing output
    print("No output found in raw_response.")
