from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Performs a search over the internet based on the provided query and returns the relevant result.

    :param query: The search term or query used for the operation.
    :type query: str
    :return: The result of the search operation as a string.
    :rtype: str
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for job postings for IoT Engineer (Python, Edge-to-Cloud Systems) in the Tadworth area and list all their details")})
    print(result)


if __name__ == "__main__":
    main()
