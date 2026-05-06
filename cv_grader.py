import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

load_dotenv()

def load_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def run_cv_grading_and_amendment():
    # 1. Load CV
    cv_path = "Mauro_Saderi CV.pdf"
    loader = PyPDFLoader(cv_path)
    cv_docs = loader.load()
    cv_content = "\n".join([doc.page_content for doc in cv_docs])

    # 2. Load Job Description
    jd_path = "job-description.txt"
    jd_content = load_text_file(jd_path)

    # 2.1 Load Skills
    skills_path = "skills.md"
    skills_content = load_text_file(skills_path)

    # 3. Pull Prompt from LangSmith (Gemini Grader)
    client = Client()
    grader_prompt = client.pull_prompt(
        "vapourisation/parse-cv-and-grade",
        secrets={"GOOGLE_API_KEY": os.environ.get("GEMINI_API_KEY", "")}
    )

    # Modify the prompt to add amendment suggestions and skills context
    amendment_instruction = f"""

Additional Skills Context:
{skills_content}

Finally, suggest which amendments can be made in the CV to match the job description more closely, using the "Additional Skills Context" provided above to enrich the suggestions with relevant experiences or skills that might be missing or under-emphasized in the current CV. Please put the suggestions under a clear header: "### Suggested Amendments". """
    grader_prompt.template += amendment_instruction

    # 4. Initialize Gemini Model
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0
    )

    # 5. Create Grading Chain
    grading_chain = grader_prompt | gemini_model | StrOutputParser()

    # 6. Execute Grading Chain with Retry
    print("--- Running CV Grading and Suggestion Generation ---")
    import time
    max_retries = 3
    grading_response = None
    for attempt in range(max_retries):
        try:
            grading_response = grading_chain.invoke({"cv": cv_content, "job_description": jd_content})
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Gemini API failed after all retries. Falling back to OpenAI for grading.")
                # Fallback to OpenAI for grading
                fallback_grader_model = ChatOpenAI(
                    model="gpt-4o",
                    openai_api_key=os.environ.get("OPENAI_API_KEY"),
                    temperature=0
                )
                fallback_chain = grader_prompt | fallback_grader_model | StrOutputParser()
                grading_response = fallback_chain.invoke({"cv": cv_content, "job_description": jd_content})

    print(grading_response)

    # 7. Extract Suggested Amendments
    # We look for the "### Suggested Amendments" section
    if "### Suggested Amendments" in grading_response:
        suggestions = grading_response.split("### Suggested Amendments")[-1].strip()
    else:
        # Fallback if the model didn't use the exact header
        suggestions = grading_response

    # 8. Initialize OpenAI Model
    openai_model = ChatOpenAI(
        model="gpt-4o",
        #api_key=os.environ.get("OPENAI_API_KEY", ""),
        temperature=0
    )

    # 9. Create Amendment Prompt
    amendment_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert CV writer. 
        Your task is to review my resume based on a job description to make it a perfect match. 
        Rewrite weak bullet points. Add missing keywords from the job description. 
        Flag anything that would get filtered by ATS.
        List the skills I am missing. Tell me which ones I can reframe from existing experience and which ones I need to develop.   
        Use informaiton the extra skills list to enhance the resume"""),
        ("user", "Here is the original CV:\n\n{cv}\n\nHere are the extra skills list :\n\n{suggestions}\n\nHere is the job description for reference:\n\n{job_description}\n\nPlease provide the full amended CV.")
    ])

    # 10. Create Amendment Chain
    amendment_chain = amendment_prompt | openai_model | StrOutputParser()

    # 11. Execute Amendment Chain
    print("\n--- Running CV Amendment with OpenAI ---")
    amended_cv = amendment_chain.invoke({
        "cv": cv_content,
        "suggestions": suggestions,
        "job_description": jd_content
    })

    # 12. Print Result
    print("\n--- Amended CV ---")
    print(amended_cv)

if __name__ == "__main__":
    run_cv_grading_and_amendment()
