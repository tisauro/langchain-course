import os

from dotenv import load_dotenv
from langchain_classic import text_splitter
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
load_dotenv()


def main():
    print("Hello from langchain-course!")

if __name__ == "__main__":
    main()
    filepath = Path("./blog-text.txt")
    loader = TextLoader(file_path=filepath)
    document = loader.load()
    print(document)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"crated {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
