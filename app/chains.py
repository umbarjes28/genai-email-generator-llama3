import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
    """
    ### ROLE REQUIREMENTS:
    {job_description}
    
    ### TASK:
    You are John, a Business Development Executive at Google — an AI and Software Consulting firm committed to enhancing 
    business workflows through intelligent automation solutions.
    With a proven track record, Google has supported numerous businesses in scaling operations, improving efficiency, 
    streamlining processes, and lowering costs through tailored technology strategies.
    
    Your objective is to craft a compelling cold email to the client regarding the opportunity listed above. The email 
    should highlight how google’s services align with their needs and demonstrate our capabilities.
    Incorporate the most relevant references from the following portfolio links to strengthen the message: {link_list}
    
    Stay in character as Jonh, BDE at google. Avoid any introductory commentary or explanations.
    
    ### COLD EMAIL (START DIRECTLY):
    
    """
)
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))