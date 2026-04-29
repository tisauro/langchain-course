from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for the agent response
    """
    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of source used to generate the answers")


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(
        content="Search for remote job postings for IoT Engineer (Python, Edge-to-Cloud Systems) and list all their details in UK and EU")})
    print(result)


if __name__ == "__main__":
    main()
