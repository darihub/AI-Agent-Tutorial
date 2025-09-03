from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import PyPDF2

load_dotenv() # Load environment variables from a .env file

# LLM Setup

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse) # parses raw LLM output into structured ResearchResponse

# Load documentation from PDF
with open("docs/pos_docs.pdf", "rb") as doc_file:
    reader = PyPDF2.PdfReader(doc_file)
    documentation = ""
    for page in reader.pages:
        documentation += page.extract_text() or "" # Extract text from each page


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are an assistant that helps the user utilize and navigate a POS system.
            Respond either in Spanish or English, depending on the user's input.
            You will assist according to the system's and other source's documentation.
            Here is the documentation you should use:
            {documentation}
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
