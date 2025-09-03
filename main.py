from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv() # Load environment variables from a .env file

# LLM Setup
llm = ChatOpenAI(model="gpt-4o")