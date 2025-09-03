from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv() # Load environment variables from a .env file

# LLM Setup

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_template(
    [
        (
            "system",
            """
            You are an assistant that helps the user utilize and navigate a POS system.
            You will assist according to the system's and other source's documentation."
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]

).partial(format_instructions=parser.get_format_instructions()) # partially fill prompt with format instructions