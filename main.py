import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo")
vector_store = PineconeVectorStore(embedding=embeddings, index_name=os.environ.get("INDEX_NAME"))
retriever = vector_store.as_retriever(search_kargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_retrieval_chain_with_LCEL():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language)
    Returns a chain that can be invoked with {"question": "..."}
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(itemgetter("question")) | retriever | RunnableLambda(format_docs)
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return  retrieval_chain


if __name__ == "__main__":
    question = "Which vector database is best for production environment?"

    # result_raw = llm.invoke([HumanMessage(content=question)])
    # print(f"Without RAG: {result_raw.content}")

    chain_with_lcel = create_retrieval_chain_with_LCEL()
    result = chain_with_lcel.invoke({"question": question})
    print(result)

