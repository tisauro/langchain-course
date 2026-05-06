import os

from dotenv import load_dotenv
from langsmith import Client


load_dotenv()




if __name__=="__main__":
    client = Client()
    prompt = client.pull_prompt(
        "vapourisation/parse-cv-and-grade",
        include_model=True,
        secrets={"GOOGLE_API_KEY": os.environ.get("GEMINI_API_KEY", "")}
    )

    print(prompt)